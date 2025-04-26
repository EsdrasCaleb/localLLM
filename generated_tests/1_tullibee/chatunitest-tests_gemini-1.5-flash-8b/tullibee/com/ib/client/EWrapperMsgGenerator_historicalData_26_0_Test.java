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

class EWrapperMsgGenerator_historicalData_26_0_Test {

    @Test
    void testHistoricalData_validInput() {
        int reqId = 123;
        String date = "2023-10-27";
        double open = 100.50;
        double high = 101.25;
        double low = 99.75;
        double close = 100.00;
        int volume = 1000;
        int count = 5;
        double WAP = 100.10;
        boolean hasGaps = true;
        String expectedOutput = "id=123 date = 2023-10-27 open=100.5 high=101.25 low=99.75 close=100.0 volume=1000 count=5 WAP=100.1 hasGaps=true";
        String actualOutput = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testHistoricalData_zeroVolume() {
        int reqId = 456;
        String date = "2023-10-28";
        double open = 150.50;
        double high = 151.25;
        double low = 149.75;
        double close = 150.00;
        int volume = 0;
        int count = 2;
        double WAP = 150.10;
        boolean hasGaps = false;
        String expectedOutput = "id=456 date = 2023-10-28 open=150.5 high=151.25 low=149.75 close=150.0 volume=0 count=2 WAP=150.1 hasGaps=false";
        String actualOutput = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testHistoricalData_nullDate() {
        int reqId = 789;
        String date = null;
        double open = 200.50;
        double high = 201.25;
        double low = 199.75;
        double close = 200.00;
        int volume = 1500;
        int count = 8;
        double WAP = 200.10;
        boolean hasGaps = true;
        String expectedOutput = "id=789 date = null open=200.5 high=201.25 low=199.75 close=200.0 volume=1500 count=8 WAP=200.1 hasGaps=true";
        String actualOutput = EWrapperMsgGenerator.historicalData(reqId, date, open, high, low, close, volume, count, WAP, hasGaps);
        assertEquals(expectedOutput, actualOutput);
    }
}
