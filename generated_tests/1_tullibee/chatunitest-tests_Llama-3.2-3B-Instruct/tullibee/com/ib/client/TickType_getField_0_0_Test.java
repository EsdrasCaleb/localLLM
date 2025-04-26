package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TickType_getField_0_0_Test {

    @Test
    public void testGetField_BID_SIZE() {
        String result = TickType.getField(TickType.BID_SIZE);
        assertEquals("bidSize", result);
    }

    @Test
    public void testGetField_BID() {
        String result = TickType.getField(TickType.BID);
        assertEquals("bidPrice", result);
    }

    @Test
    public void testGetField_ASK() {
        String result = TickType.getField(TickType.ASK);
        assertEquals("askPrice", result);
    }

    @Test
    public void testGetField_LAST() {
        String result = TickType.getField(TickType.LAST);
        assertEquals("lastPrice", result);
    }

    @Test
    public void testGetField_UNKNOWN() {
        String result = TickType.getField(100);
        assertEquals("unknown", result);
    }
}
