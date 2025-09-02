---
description: Enforces strict production-ready development protocols with zero tolerance for shortcuts or placeholders
---

You MUST adhere to the following strict production development protocols:

## Implementation Standards
- **No placeholders**: Every function, class, and module must be fully implemented with production-ready code
- **No mock data**: All data handling must use real schemas, validation, and proper data structures
- **No TODO comments**: Complete all implementations before marking tasks as done
- **No hardcoded values**: Use proper configuration management and environment variables

## Code Quality Requirements
- **Error handling**: Every function must include comprehensive error handling with specific error types
- **Input validation**: Validate all inputs with proper type checking and boundary validation
- **Edge cases**: Handle null values, empty collections, network failures, and unexpected states
- **Resource cleanup**: Properly dispose of resources, close connections, and prevent memory leaks

## Architecture and Design
- **Design patterns**: Use appropriate patterns (Factory, Observer, Strategy, etc.) where beneficial
- **SOLID principles**: Follow Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion
- **Separation of concerns**: Keep business logic, data access, and presentation layers distinct
- **Dependency injection**: Use proper DI patterns to avoid tight coupling

## Testing Requirements
- **Unit tests**: Write tests for all public methods and critical private methods
- **Integration tests**: Test component interactions and external dependencies
- **Edge case testing**: Include tests for error conditions, boundary values, and failure scenarios
- **Test coverage**: Aim for meaningful coverage that tests behavior, not just lines of code

## Documentation and Maintenance
- **API documentation**: Document all public interfaces with clear examples
- **Code comments**: Explain complex business logic and non-obvious implementations
- **Type annotations**: Use strong typing throughout the codebase
- **Logging**: Include appropriate logging levels for debugging and monitoring

## Performance and Security
- **Performance considerations**: Avoid N+1 queries, implement caching where appropriate, optimize algorithms
- **Security best practices**: Sanitize inputs, use parameterized queries, implement proper authentication/authorization
- **Scalability**: Design components to handle increased load and data volume

## Verification Checklist
Before completing any task, verify:
1. All code is production-ready with no placeholders
2. Error handling covers all failure scenarios
3. Input validation is comprehensive
4. Tests are written and passing
5. Documentation is complete
6. Security considerations are addressed
7. Performance implications are considered

REJECT any implementation that includes placeholders, TODOs, or shortcuts. Demand complete, production-ready solutions that can be deployed immediately without further modification.