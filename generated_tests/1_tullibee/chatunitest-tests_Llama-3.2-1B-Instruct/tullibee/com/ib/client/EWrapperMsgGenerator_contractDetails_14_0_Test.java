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

public class EWrapperMsgGenerator_contractDetails_14_0_Test {

    @Test
    public void testContractDetails() {
        // Arrange
        ContractDetails contractDetails = new ContractDetails();
        int reqId = 1;
        // Act
        String msg = EWrapperMsgGenerator.contractDetails(reqId, contractDetails);
        // Assert
        assertEquals("Contract details message", msg);
    }
}
