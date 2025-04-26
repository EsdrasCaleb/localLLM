package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testErrorNormalFlow() {
        int id = 1;
        int errorCode = 123;
        String errorMsg = "Something went wrong";
        String expectedError = "1 | 123 | Something went wrong";
        String actualError = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expectedError, actualError);
    }

    @Test
    public void testErrorInvalidId() {
        int id = 0;
        int errorCode = 123;
        String errorMsg = "Something went wrong";
        String expectedError = "0 | 123 | Something went wrong";
        String actualError = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expectedError, actualError);
    }

    @Test
    public void testErrorInvalidErrorCode() {
        int id = 1;
        int errorCode = 0;
        String errorMsg = "Something went wrong";
        String expectedError = "1 | 0 | Something went wrong";
        String actualError = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expectedError, actualError);
    }

    @Test
    public void testErrorEmptyErrorMsg() {
        int id = 1;
        int errorCode = 123;
        String errorMsg = "";
        String expectedError = "1 | 123 | ";
        String actualError = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expectedError, actualError);
    }

    @Test
    public void testErrorNullErrorMsg() {
        int id = 1;
        int errorCode = 123;
        String errorMsg = null;
        assertThrows(NullPointerException.class, () -> AnyWrapperMsgGenerator.error(id, errorCode, errorMsg));
    }
}
