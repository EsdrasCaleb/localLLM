package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickString_4_4_Test {

    @Test
    public void testTickString() {
        int tickerId = 1;
        int tickType = TickType.BID_SIZE;
        String value = "100";
        String expected = "id=1  BID_SIZE=100";
        String actual = EWrapperMsgGenerator.tickString(tickerId, tickType, value);
        assertEquals(expected, actual);
    }
}
