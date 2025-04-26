package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_execDetailsEnd_20_0_Test {

    @ParameterizedTest
    @CsvSource({ "1", "10", "100" })
    void testExecDetailsEnd(int reqId) {
        String expected = "reqId = " + reqId + " =============== end ===============";
        String actual = EWrapperMsgGenerator.execDetailsEnd(reqId);
        assertEquals(expected, actual);
    }
}
