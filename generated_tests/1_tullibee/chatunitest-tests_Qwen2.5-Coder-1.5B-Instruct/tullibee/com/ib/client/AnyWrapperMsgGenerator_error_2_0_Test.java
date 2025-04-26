package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    void testError() {
        // Create a mock object for the AnyWrapperMsgGenerator class
        AnyWrapperMsgGenerator anyWrapperMsgGeneratorMock = Mockito.mock(AnyWrapperMsgGenerator.class);
        // Call the error method on the mock object with provided arguments
        String result = anyWrapperMsgGeneratorMock.error(101, 404, "Resource not found");
        // Verify the result using assertEquals method
        assertEquals("101 | 404 | Resource not found", result);
    }
}
