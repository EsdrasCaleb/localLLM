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

public class EWrapperMsgGenerator_tickEFP_5_0_Test {

    @Test
    public void testTickEFP() {
        int tickerId = 123;
        int tickType = 2;
        double basisPoints = 10.0;
        String formattedBasisPoints = "10.0";
        double impliedFuture = 15.0;
        int holdDays = 20;
        String futureExpiry = "2020-01-01";
        double dividendImpact = 0.5;
        double dividendsToExpiry = 0.7;
        String expected = "id=123  TICK_TYPE: basisPoints = 10.0/10.0 impliedFuture = 15.0 holdDays = 20 futureExpiry = 2020-01-01 dividendImpact = 0.5 dividends to expiry = 0.7";
        String result = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        assertEquals(expected, result);
    }
}
