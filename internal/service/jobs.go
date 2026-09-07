package service

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

var ErrBusy = errors.New("another background job is running")

type Job struct {
	ID       string     `json:"id"`
	Kind     string     `json:"kind"`
	Status   string     `json:"status"`
	Done     int        `json:"done"`
	Total    int        `json:"total"`
	Detail   string     `json:"detail,omitempty"`
	Created  time.Time  `json:"created_at"`
	Finished *time.Time `json:"finished_at,omitempty"`
	Result   any        `json:"result,omitempty"`
	Error    string     `json:"error,omitempty"`
}
type jobState struct {
	job    Job
	cancel context.CancelFunc
}
type Jobs struct {
	mu     sync.Mutex
	ctx    context.Context
	jobs   map[string]*jobState
	active string
	wg     sync.WaitGroup
}

func NewJobs(ctx context.Context) *Jobs { return &Jobs{ctx: ctx, jobs: map[string]*jobState{}} }
func (j *Jobs) Start(kind string, run func(context.Context, Progress) (any, error)) (Job, error) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if err := j.ctx.Err(); err != nil {
		return Job{}, err
	}
	if j.active != "" {
		return Job{}, ErrBusy
	}
	// Bound retained completed jobs. Durable JSONL outputs are independent of this registry.
	if len(j.jobs) >= 100 {
		var oldest *jobState
		for _, s := range j.jobs {
			if oldest == nil || s.job.Created.Before(oldest.job.Created) {
				oldest = s
			}
		}
		delete(j.jobs, oldest.job.ID)
	}
	ctx, cancel := context.WithCancel(j.ctx)
	state := &jobState{job: Job{ID: uuid.NewString(), Kind: kind, Status: "running", Created: time.Now().UTC()}, cancel: cancel}
	j.jobs[state.job.ID] = state
	j.active = state.job.ID
	initial := state.job
	j.wg.Add(1)
	go func() {
		defer j.wg.Done()
		defer cancel()
		var result any
		var err error
		defer func() {
			if v := recover(); v != nil {
				err = fmt.Errorf("job panicked: %v", v)
			}
			j.mu.Lock()
			defer j.mu.Unlock()
			now := time.Now().UTC()
			state.job.Finished = &now
			state.job.Result = result
			state.job.Status = "completed"
			if err != nil {
				state.job.Status = "failed"
				state.job.Error = err.Error()
				if errors.Is(err, context.Canceled) {
					state.job.Status = "cancelled"
				}
			}
			j.active = ""
		}()
		result, err = run(ctx, func(done, total int, detail string) {
			j.mu.Lock()
			defer j.mu.Unlock()
			state.job.Done = done
			state.job.Total = total
			state.job.Detail = detail
		})
	}()
	return initial, nil
}
func (j *Jobs) Get(id string) (Job, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	s, ok := j.jobs[id]
	if !ok {
		return Job{}, false
	}
	return s.job, true
}
func (j *Jobs) Cancel(id string) bool {
	j.mu.Lock()
	defer j.mu.Unlock()
	s, ok := j.jobs[id]
	if ok && s.job.Status == "running" {
		s.cancel()
	}
	return ok
}
func (j *Jobs) Wait() { j.wg.Wait() }
