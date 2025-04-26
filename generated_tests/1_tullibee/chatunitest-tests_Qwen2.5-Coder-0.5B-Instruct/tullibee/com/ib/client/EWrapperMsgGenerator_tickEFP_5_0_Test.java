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

class EWrapperMsgGenerator_tickEFP_5_0_Test {

    @Test
    void testTickEFP() {
        // Arrange
        int tickerId = 12345;
        int tickType = 0;
        double basisPoints = 10.5;
        String formattedBasisPoints = "10.5";
        double impliedFuture = 100.0;
        int holdDays = 10;
        String futureExpiry = "2023-10-01";
        double dividendImpact = 0.0;
        double dividendsToExpiry = 0.0;
        // Act
        String result = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        assertEquals("id=12345  BID: 10.5 / 10.5 impliedFuture = 100.0 holdDays = 10 futureExpiry = 2023-10-01 dividendImpact = 0.0 dividends to expiry = 0.0", result);
    }
}
