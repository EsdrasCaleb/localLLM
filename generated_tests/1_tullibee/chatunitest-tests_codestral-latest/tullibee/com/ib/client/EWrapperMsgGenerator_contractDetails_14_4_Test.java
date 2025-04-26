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

class EWrapperMsgGenerator_contractDetails_14_4_Test {

    private ContractDetails contractDetails;

    private Contract contract;

    @BeforeEach
    void setUp() {
        contractDetails = mock(ContractDetails.class);
        contract = mock(Contract.class);
        contractDetails.m_summary = contract;
    }

    @Test
    void testContractDetails() {
        int reqId = 1;
        when(contract.toString()).thenReturn("Contract details");
        when(contractDetails.toString()).thenReturn("Additional contract details");
        String expectedMsg = "reqId = 1 ===================================\n" + " ---- Contract Details begin ----\n" + "Contract detailsAdditional contract details" + " ---- Contract Details End ----\n";
        String actualMsg = EWrapperMsgGenerator.contractDetails(reqId, contractDetails);
        assertEquals(expectedMsg, actualMsg);
    }
}
