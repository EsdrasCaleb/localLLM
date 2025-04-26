// Test method
package com.ib.client;

import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class EWrapperMsgGenerator_tickEFP_5_0_Test {

    @Mock
    EWrapperMsgGenerator eWrapperMsgGenerator;

    @Test
    public void testTickEFP() {
        // Arrange
        int tickerId = 1;
        int tickType = 2;
        double basisPoints = 100.0;
        String formattedBasisPoints = "100.0";
        double impliedFuture = 50.0;
        int holdDays = 30;
        String futureExpiry = "2023-01-01";
        double dividendImpact = 10.0;
        double dividendsToExpiry = 20.0;
        String expected = "id=" + tickerId + "  " + TickType.getField(tickType) + ": basisPoints = " + basisPoints + "/" + formattedBasisPoints + " impliedFuture = " + impliedFuture + " holdDays = " + holdDays + " futureExpiry = " + futureExpiry + " dividendImpact = " + dividendImpact + " dividends to expiry = " + dividendsToExpiry;
        // Act
        String actual = eWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        assertEquals(expected, actual);
    }
}
