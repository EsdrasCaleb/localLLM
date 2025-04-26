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

public class EWrapperMsgGenerator_contractDetails_14_1_Test {

    @Test
    void testContractDetails() {
        int reqId = 1;
        ContractDetails contractDetails = new ContractDetails();
        contractDetails.m_summary = new Contract();
        String result = EWrapperMsgGenerator.contractDetails(reqId, contractDetails);
        assertEquals("reqId = 1 ===================================\n" + " ---- Contract Details begin ----\n" + " ---- Contract Details End ----\n", result);
    }
}
