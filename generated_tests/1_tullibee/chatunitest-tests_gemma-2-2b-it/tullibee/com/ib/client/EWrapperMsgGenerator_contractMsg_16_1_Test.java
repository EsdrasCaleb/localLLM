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

public class EWrapperMsgGenerator_contractMsg_16_1_Test {

    @Test
    void contractMsg() {
        // Arrange
        Contract contract = mock(Contract.class);
        // Act
        String msg = EWrapperMsgGenerator.contractMsg(contract);
        // Assert
        assertEquals("conid = 12345\n symbol = AAPL\n secType = OPTIONS\n expiry = 2023-12-31\n strike = 150\n right = NONE\n multiplier = 1\n exchange = NYSE\n primaryExch = NYSE\n currency = USD\n localSymbol = AAPL\n", msg);
    }
}
