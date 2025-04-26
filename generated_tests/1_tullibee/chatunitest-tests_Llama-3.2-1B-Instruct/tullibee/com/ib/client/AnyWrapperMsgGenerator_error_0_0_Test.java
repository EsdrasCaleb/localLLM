package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.function.Supplier;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_0_0_Test {

    @Mock
    private Supplier<String> exceptionSupplier;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @Test
    public void testError() {
        // Arrange
        Exception exception = new Exception("Mocked Exception");
        String expectedOutput = "Error - Mocked Exception";
        // Act
        String actualOutput = anyWrapperMsgGenerator.error(exception);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
