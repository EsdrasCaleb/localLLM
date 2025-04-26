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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickEFP_5_0_Test {

    // Test class
    @Test
    public void testTickEFP() {
        int tickerId = 1;
        int tickType = 1;
        double basisPoints = 1.0;
        String formattedBasisPoints = "1.0";
        double impliedFuture = 1.0;
        int holdDays = 1;
        String futureExpiry = "1.0";
        double dividendImpact = 1.0;
        double dividendsToExpiry = 1.0;
        String expected = "id=1  TickType: basisPoints = 1.0/1.0 impliedFuture = 1.0 holdDays = 1 futureExpiry = 1.0 dividendImpact = 1.0 dividends to expiry = 1.0";
        String actual = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        assertEquals(expected, actual);
    }
}
