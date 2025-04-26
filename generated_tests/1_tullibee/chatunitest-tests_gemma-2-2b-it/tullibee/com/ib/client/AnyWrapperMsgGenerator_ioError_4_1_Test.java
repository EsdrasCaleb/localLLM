package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_ioError_4_1_Test {

    @Test
    void ioError_ShouldReturnErrorString() {
        // Arrange
        Exception ex = new Exception("Test Exception");
        AnyWrapperMsgGenerator anyWrapperMsgGenerator = new AnyWrapperMsgGenerator();
        // Act
        String result = anyWrapperMsgGenerator.ioError(ex);
        // Assert
        assertEquals("error(Test Exception)", result);
    }
}
