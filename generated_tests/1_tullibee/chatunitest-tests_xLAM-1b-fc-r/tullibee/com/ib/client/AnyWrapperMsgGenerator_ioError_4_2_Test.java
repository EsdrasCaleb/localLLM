package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_ioError_4_2_Test {

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testIoError() {
        Exception exception = mock(Exception.class);
        String expectedMessage = "Error occurred";
        when(exception.getMessage()).thenReturn(expectedMessage);
        String result = anyWrapperMsgGenerator.ioError(exception);
        assertEquals(expectedMessage, result);
    }
}
