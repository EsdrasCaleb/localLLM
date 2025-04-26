package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    void testError() {
        // <Buggy Line>: incompatible types: int cannot be converted to java.lang.String
        int id = 123;
        int errorCode = 456;
        String errorMessage = "An error occurred";
        // <Buggy Line>: incompatible types: java.lang.String cannot be converted to int
        String errorMsg = AnyWrapperMsgGenerator.error(id, errorCode, errorMessage);
        // Act
        String actual = errorMsg;
        // Assert
        assertEquals("123 | 456 | An error occurred", actual);
    }
}
