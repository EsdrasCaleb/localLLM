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

public class EWrapperMsgGenerator_tickEFP_5_1_Test {

    @Test
    public void testTickEFP() {
        int tickerId = 1;
        int tickType = 2;
        double basisPoints = 3.0;
        String formattedBasisPoints = "3.0%";
        double impliedFuture = 4.0;
        int holdDays = 5;
        String futureExpiry = "2023-12-31";
        double dividendImpact = 6.0;
        double dividendsToExpiry = 7.0;
        String expected = "id=1  2: basisPoints = 3.0/3.0% impliedFuture = 4.0 holdDays = 5 futureExpiry = 2023-12-31 dividendImpact = 6.0 dividends to expiry = 7.0";
        String result = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        assertEquals(expected, result);
    }
}
