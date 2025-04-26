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

class EWrapperMsgGenerator_fundamentalData_32_0_Test {

    @Test
    void fundamentalData_emptyData_returnsCorrectString() {
        int reqId = 456;
        String data = "";
        String expected = "id  = 456 len = 0\n";
        String actual = EWrapperMsgGenerator.fundamentalData(reqId, data);
        assertEquals(expected, actual);
    }
}
