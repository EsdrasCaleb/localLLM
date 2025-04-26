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

class EWrapperMsgGenerator_deltaNeutralValidation_33_0_Test {

    @Test
    void testDeltaNeutralValidation() {
        // Arrange
        int reqId = 123;
        UnderComp underComp = new UnderComp();
        underComp.m_conId = 456;
        underComp.m_delta = 0.01;
        underComp.m_price = 100.0;
        // Act
        String result = EWrapperMsgGenerator.deltaNeutralValidation(reqId, underComp);
        // Assert
        assertEquals("id = 123 underComp.conId = 456 underComp.delta = 0.01 underComp.price = 100.0", result);
    }
}
