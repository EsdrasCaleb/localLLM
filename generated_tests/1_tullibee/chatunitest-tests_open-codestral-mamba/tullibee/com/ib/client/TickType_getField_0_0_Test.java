package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TickType_getField_0_0_Test {

    @Test
    public void testGetField() {
        // Test case for BID_SIZE
        assertEquals("bidSize", TickType.getField(TickType.BID_SIZE));
        // Test case for BID
        assertEquals("bidPrice", TickType.getField(TickType.BID));
        // Test case for ASK
        assertEquals("askPrice", TickType.getField(TickType.ASK));
        // Test case for ASK_SIZE
        assertEquals("askSize", TickType.getField(TickType.ASK_SIZE));
        // Test case for LAST
        assertEquals("lastPrice", TickType.getField(TickType.LAST));
        // Test case for LAST_SIZE
        assertEquals("lastSize", TickType.getField(TickType.LAST_SIZE));
        // Test case for HIGH
        assertEquals("high", TickType.getField(TickType.HIGH));
        // Test case for LOW
        assertEquals("low", TickType.getField(TickType.LOW));
        // Test case for VOLUME
        assertEquals("volume", TickType.getField(TickType.VOLUME));
        // Test case for CLOSE
        assertEquals("close", TickType.getField(TickType.CLOSE));
        // Test case for BID_OPTION
        assertEquals("bidOptComp", TickType.getField(TickType.BID_OPTION));
        // Test case for ASK_OPTION
        assertEquals("askOptComp", TickType.getField(TickType.ASK_OPTION));
        // Test case for LAST_OPTION
        assertEquals("lastOptComp", TickType.getField(TickType.LAST_OPTION));
        // Test case for MODEL_OPTION
        assertEquals("modelOptComp", TickType.getField(TickType.MODEL_OPTION));
        // Test case for OPEN
        assertEquals("open", TickType.getField(TickType.OPEN));
        // Test case for LOW_13_WEEK
        assertEquals("13WeekLow", TickType.getField(TickType.LOW_13_WEEK));
        // Test case for HIGH_13_WEEK
        assertEquals("13WeekHigh", TickType.getField(TickType.HIGH_13_WEEK));
        // Test case for LOW_26_WEEK
        assertEquals("26WeekLow", TickType.getField(TickType.LOW_26_WEEK));
        // Test case for HIGH_26_WEEK
        assertEquals("26WeekHigh", TickType.getField(TickType.HIGH_26_WEEK));
        // Test case for LOW_52_WEEK
        assertEquals("52WeekLow", TickType.getField(TickType.LOW_52_WEEK));
        // Test case for HIGH_52_WEEK
        assertEquals("52WeekHigh", TickType.getField(TickType.HIGH_52_WEEK));
        // Test case for AVG_VOLUME
        assertEquals("AvgVolume", TickType.getField(TickType.AVG_VOLUME));
        // Test case for OPEN_INTEREST
        assertEquals("OpenInterest", TickType.getField(TickType.OPEN_INTEREST));
        // Test case for OPTION_HISTORICAL_VOL
        assertEquals("OptionHistoricalVolatility", TickType.getField(TickType.OPTION_HISTORICAL_VOL));
        // Test case for OPTION_IMPLIED_VOL
        assertEquals("OptionImpliedVolatility", TickType.getField(TickType.OPTION_IMPLIED_VOL));
        // Test case for OPTION_BID_EXCH
        assertEquals("OptionBidExchStr", TickType.getField(TickType.OPTION_BID_EXCH));
        // Test case for OPTION_ASK_EXCH
        assertEquals("OptionAskExchStr", TickType.getField(TickType.OPTION_ASK_EXCH));
    }
}
