package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    void error() {
        // Given
        int id = 123;
        int errorCode = 456;
        String errorMsg = "This is an error message";
        // When
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Then
        assertEquals("123 | 456 | This is an error message", result);
    }
}
