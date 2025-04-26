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
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int tickerId = 123;
        int tickType = 1;
        double basisPoints = 10.0;
        String formattedBasisPoints = "10.0";
        double impliedFuture = 20.0;
        int holdDays = 30;
        String futureExpiry = "2022-09-30";
        double dividendImpact = 0.5;
        double dividendsToExpiry = 0.2;
        // Act
        String result = eWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        assertEquals("id=123  TICK: basisPoints = 10.0/10.0 impliedFuture = 20.0 holdDays = 30 futureExpiry = 2022-09-30 dividendImpact = 0.5 dividends to expiry = 0.2", result);
    }
}
