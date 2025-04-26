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

public class EWrapperMsgGenerator_tickGeneric_3_0_Test {

    @Test
    public void testTickGeneric() {
        int tickerId = 1;
        int tickType = 1;
        double value = 10.0;
        String expected = "id=1  BID=10.0";
        String actual = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        assertEquals(expected, actual);
    }

    @Test
    public void testTickGeneric_SCANNER_PARAMETERS() {
        int tickerId = 1;
        int tickType = 1;
        double value = 10.0;
        String expected = "id=1  SCANNER PARAMETERS:BID=10.0";
        String actual = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        assertEquals(expected, actual);
    }

    @Test
    public void testTickGeneric_FINANCIAL_ADVISOR() {
        int tickerId = 1;
        int tickType = 1;
        double value = 10.0;
        String expected = "id=1  FA:BID=10.0";
        String actual = EWrapperMsgGenerator.tickGeneric(tickerId, tickType, value);
        assertEquals(expected, actual);
    }
}
