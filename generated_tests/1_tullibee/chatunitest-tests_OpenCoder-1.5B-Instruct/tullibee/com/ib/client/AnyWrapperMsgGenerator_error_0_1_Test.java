package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_0_1_Test {

    @Mock
    private Exception exception;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testError() {
        when(exception.getMessage()).thenReturn("Exception message");
        String result = anyWrapperMsgGenerator.error(exception);
        assertEquals("Error - Exception message", result);
    }
}
