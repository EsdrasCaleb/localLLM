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
        // Arrange
        int tickerId = 1;
        // Assuming 2 corresponds to a valid TickType
        int tickType = 2;
        double basisPoints = 100.5;
        String formattedBasisPoints = "100.5%";
        double impliedFuture = 1500.75;
        int holdDays = 30;
        String futureExpiry = "2023-12-31";
        double dividendImpact = 2.5;
        double dividendsToExpiry = 1.0;
        // Act
        String result = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        String expected = "id=1  " + TickType.getField(tickType) + ": basisPoints = 100.5/100.5% impliedFuture = 1500.75 holdDays = 30" + " futureExpiry = 2023-12-31 dividendImpact = 2.5 dividends to expiry = 1.0";
        assertEquals(expected, result);
    }
}
