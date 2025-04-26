package com.ib.client;

import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Test
    public void testIoError() throws Exception {
        // Mock an IOException
        IOException mockException = mock(IOException.class);
        // Define the behavior of the mocked exception
        when(mockException.getMessage()).thenReturn("Mocked IO Exception");
        // Invoke the method under test
        String result = AnyWrapperMsgGenerator.ioError(mockException);
        // Verify the result
        assertEquals("Mocked IO Exception", result);
    }
}
