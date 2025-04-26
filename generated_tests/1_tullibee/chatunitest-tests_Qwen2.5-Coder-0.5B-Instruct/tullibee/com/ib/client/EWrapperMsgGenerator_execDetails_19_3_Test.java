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

class EWrapperMsgGenerator_execDetails_19_3_Test {

    @Test
    void testExecDetails() {
        // Arrange
        Contract mockContract = Mockito.mock(Contract.class);
        Execution mockExecution = Mockito.mock(Execution.class);
        // Act
        String result = EWrapperMsgGenerator.execDetails(12345, mockContract, mockExecution);
        // Assert
        assertEquals(" ---- Execution Details begin ----\n" + "reqId = 12345\n" + "orderId = 0\n" + "clientId = 0\n" + "symbol = XYZ\n" + "secType = SPX\n" + "expiry = 2023-09-01\n" + "strike = 100.00\n" + "right = CALL\n" + "contractExchange = NASDAQ\n" + "currency = USD\n" + "localSymbol = XYZ\n" + "execId = 0\n" + "time = 1234567890\n" + "acctNumber = 1234567890\n" + "executionExchange = NASDAQ\n" + "side = BUY\n" + "shares = 1000\n" + "price = 100.00\n" + "permId = 0\n" + "liquidation = false\n" + "cumQty = 0\n" + "avgPrice = 100.00\n" + " ---- Execution Details end ----\n", result);
    }
}
