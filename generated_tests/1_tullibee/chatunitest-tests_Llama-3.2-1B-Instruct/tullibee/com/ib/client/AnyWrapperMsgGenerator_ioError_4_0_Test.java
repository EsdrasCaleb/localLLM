package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.function.Function;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_ioError_4_0_Test {

    @Mock
    private Exception exception;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testIoError() {
        // Arrange
        String expected = "IO Error: ";
        when(exception.getMessage()).thenReturn(expected);
        // Act
        String actual = anyWrapperMsgGenerator.ioError(exception);
        // Assert
        assertEquals(expected, actual);
    }
}
