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

public class EWrapperMsgGenerator_contractDetailsEnd_18_2_Test {

    @Test
    public void testContractDetailsEnd() {
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int reqId = 12345;
        String expected = "reqId = " + reqId + " =============== end ===============";
        String result = generator.contractDetailsEnd(reqId);
        assertEquals(expected, result);
    }
}
