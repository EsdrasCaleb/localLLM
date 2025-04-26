package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_2_3_Test {

    @Test
    public void testErrorMethod() {
        String expected = "1 | 404 | Resource not found";
        String actual = AnyWrapperMsgGenerator.error(1, 404, "Resource not found");
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorMethodIdZero() {
        String expected = "0 | 404 | Resource not found";
        String actual = AnyWrapperMsgGenerator.error(0, 404, "Resource not found");
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorMethodIdNegative() {
        String expected = "0 | 404 | Resource not found";
        String actual = AnyWrapperMsgGenerator.error(-1, 404, "Resource not found");
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorMethodErrorCodeZero() {
        String expected = "1 | 0 | Resource not found";
        String actual = AnyWrapperMsgGenerator.error(1, 0, "Resource not found");
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorMethodErrorCodeNegative() {
        String expected = "1 | 0 | Resource not found";
        String actual = AnyWrapperMsgGenerator.error(1, -1, "Resource not found");
        assertEquals(expected, actual);
    }

    @Test
    public void testErrorMethodEmptyErrorMsg() {
        String expected = "1 | 404 | ";
        String actual = AnyWrapperMsgGenerator.error(1, 404, "");
        assertEquals(expected, actual);
    }
}
