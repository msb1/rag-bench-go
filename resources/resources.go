package resources

import _ "embed"

//go:embed openapi.json
var OpenAPI []byte

//go:embed swagger.html
var Swagger string
