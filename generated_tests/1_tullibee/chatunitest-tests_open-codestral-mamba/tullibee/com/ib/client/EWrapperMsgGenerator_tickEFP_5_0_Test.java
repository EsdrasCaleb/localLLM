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
        int tickerId = 1;
        int tickType = 2;
        double basisPoints = 0.5;
        String formattedBasisPoints = "0.5%";
        double impliedFuture = 0.01;
        int holdDays = 5;
        String futureExpiry = "2022-12-31";
        double dividendImpact = 0.02;
        double dividendsToExpiry = 0.005;
        String expectedOutput = "id=1  TickType: tickType = 0.5% impliedFuture = 0.01 holdDays = 5 futureExpiry = 2022-12-31 dividendImpact = 0.02 dividends to expiry = 0.005";
        String actualOutput = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        assertEquals(expectedOutput, actualOutput);
    }
}
