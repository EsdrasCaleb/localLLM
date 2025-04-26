package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_0_1_Test {

    @Test
    void testError() {
        // Given
        AnyWrapperMsgGenerator wrapper = new AnyWrapperMsgGenerator();
        Exception expectedException = new RuntimeException("Test Exception");
        // When
        String actualResult = wrapper.error(expectedException);
        // Then
        assertEquals("Error - Test Exception", actualResult);
    }
}
