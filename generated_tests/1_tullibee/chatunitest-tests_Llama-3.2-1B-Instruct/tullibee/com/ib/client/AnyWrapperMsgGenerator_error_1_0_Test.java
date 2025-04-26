package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_1_0_Test {

    @Mock
    private AnyWrapperMsgGenerator focal;

    @Test
    public void testError() {
        // Given
        String str = "Hello, World!";
        // When
        String result = focal.error(str);
        // Then
        assertEquals(str, result);
    }
}
