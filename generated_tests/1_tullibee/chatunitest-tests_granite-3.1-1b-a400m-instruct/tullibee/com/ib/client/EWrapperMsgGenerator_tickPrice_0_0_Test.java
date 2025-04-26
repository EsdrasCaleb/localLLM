package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    public void testTickPrice() {
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 1, 123.45, 1));
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 2, 123.45, 1));
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 1, 123.45, 0));
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 2, 123.45, 0));
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 1, 123.45, 1));
        assertEquals("id=12345  Field=12345  canAutoExecute", EWrapperMsgGenerator.tickPrice(12345, 2, 123.45, 1));
    }
}
