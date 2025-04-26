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
        int tickerId = 123;
        int tickType = 2;
        double basisPoints = 12.34;
        String formattedBasisPoints = "12.34";
        double impliedFuture = 12.34;
        int holdDays = 12;
        String futureExpiry = "2022-01-01";
        double dividendImpact = 0.12;
        double dividendsToExpiry = 0.12;
        // Act
        String result = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        assertEquals("id=123 FA: basisPoints = 12.34/12.34 impliedFuture = 12.34 holdDays = 12 futureExpiry = 2022-01-01 dividendImpact = 0.12 dividends to expiry = 0.12", result);
    }
}
