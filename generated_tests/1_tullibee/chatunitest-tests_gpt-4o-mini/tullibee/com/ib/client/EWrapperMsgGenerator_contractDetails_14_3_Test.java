package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_contractDetails_14_3_Test {

    @Test
    public void testContractDetails() throws Exception {
        // Arrange
        int reqId = 123;
        // Mocking Contract and ContractDetails
        Contract mockContract = mock(Contract.class);
        ContractDetails mockContractDetails = mock(ContractDetails.class);
        // Using reflection to set the private field m_summary in ContractDetails
        Field summaryField = ContractDetails.class.getDeclaredField("m_summary");
        summaryField.setAccessible(true);
        summaryField.set(mockContractDetails, mockContract);
        when(mockContractDetails.toString()).thenReturn("Mocked Contract Details");
        // Create an instance of EWrapperMsgGenerator
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        // Act
        // Assuming contractDetails is a non-static method
        String result = (String) EWrapperMsgGenerator.class.getDeclaredMethod("contractDetails", int.class, ContractDetails.class).invoke(generator, reqId, mockContractDetails);
        // Assert
        String expected = "reqId = 123 ===================================\n" + " ---- Contract Details begin ----\n" + "Mocked Contract Details" + " ---- Contract Details End ----\n";
        assertEquals(expected, result);
    }
}
