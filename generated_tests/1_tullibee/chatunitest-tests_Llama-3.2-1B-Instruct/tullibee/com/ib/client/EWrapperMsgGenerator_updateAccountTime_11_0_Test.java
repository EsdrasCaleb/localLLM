// Test method
package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.MethodSource;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_updateAccountTime_11_0_Test {

    @Test
    @ParameterizedTest
    @CsvSource({ "2022-01-01 00:00:00", "2022-01-02 00:00:00" })
    public void updateAccountTime(String timeStamp) {
        EWrapperMsgGenerator focal = new EWrapperMsgGenerator();
        String result = focal.updateAccountTime(timeStamp);
        assertEquals("updateAccountTime: 2022-01-01 00:00:00", result);
    }
}
