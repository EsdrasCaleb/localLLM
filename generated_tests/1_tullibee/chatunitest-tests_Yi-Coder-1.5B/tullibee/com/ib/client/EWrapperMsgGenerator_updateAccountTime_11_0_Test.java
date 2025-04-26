package com.ib.client;

import java.util.*;
import java.lang.reflect.*;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;

class EWrapperMsgGenerator_updateAccountTime_11_0_Test {

    @Test
    void testUpdateAccountTime() {
        String timeStamp = "12:34:56";
        String expected = "updateAccountTime: 12:34:56";
        String actual = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expected, actual);
    }

    @Test
    void testUpdateAccountTimeWithZero() {
        String timeStamp = "00:00:00";
        String expected = "updateAccountTime: 00:00:00";
        String actual = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expected, actual);
    }

    @Test
    void testUpdateAccountTimeWithNegative() {
        String timeStamp = "-12:34:56";
        String expected = "updateAccountTime: -12:34:56";
        String actual = EWrapperMsgGenerator.updateAccountTime(timeStamp);
        assertEquals(expected, actual);
    }
}
