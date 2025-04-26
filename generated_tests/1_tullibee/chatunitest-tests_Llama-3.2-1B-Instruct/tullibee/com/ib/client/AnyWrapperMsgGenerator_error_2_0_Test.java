package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Random;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Mock
    private AnyWrapperMsgGenerator anyWrapperMsgGenerator;

    @InjectMocks
    private AnyWrapperMsgGenerator anyWrapperMsgGeneratorTest;

    @Test
    public void testError() {
        // Arrange
        int id = 123;
        int errorCode = 500;
        String errorMsg = "Error message";
        // Act
        String result = anyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        // Assert
        assertEquals("123 | 500 | Error message", result);
    }
}
