package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testError() {
        int id = 1;
        int errorCode = 100;
        String errorMsg = "Test error message";
        String expected = "1 | 100 | Test error message";
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorWithZeroId() {
        int id = 0;
        int errorCode = 100;
        String errorMsg = "Test error message";
        String expected = "0 | 100 | Test error message";
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorWithZeroErrorCode() {
        int id = 1;
        int errorCode = 0;
        String errorMsg = "Test error message";
        String expected = "1 | 0 | Test error message";
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorWithEmptyErrorMsg() {
        int id = 1;
        int errorCode = 100;
        String errorMsg = "";
        String expected = "1 | 100 | ";
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorWithNullErrorMsg() {
        int id = 1;
        int errorCode = 100;
        String errorMsg = null;
        String expected = "1 | 100 | null";
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }
}
