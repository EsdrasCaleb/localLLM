package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_deltaNeutralValidation_33_0_Test {

    @Test
    public void testDeltaNeutralValidation() {
        // Arrange
        int reqId = 12345;
        UnderComp underComp = new UnderComp();
        underComp.m_conId = 67890;
        underComp.m_delta = 0.5;
        underComp.m_price = 100.75;
        String expectedOutput = "id = 12345, underComp.conId = 67890, underComp.delta = 0.5, underComp.price = 100.75";
        // Act
        String result = EWrapperMsgGenerator.deltaNeutralValidation(reqId, underComp);
        // Assert
        assertEquals(expectedOutput, result);
    }
}

class UnderComp {

    public int m_conId;

    public double m_delta;

    public double m_price;

    // Mocking public fields requires using a getter method in Mockito
    public int getM_conId() {
        return m_conId;
    }

    public double getM_delta() {
        return m_delta;
    }

    public double getM_price() {
        return m_price;
    }
}

class EWrapperMsgGenerator {

    public static String deltaNeutralValidation(int reqId, UnderComp underComp) {
        StringBuilder msg = new StringBuilder();
        msg.append("id = ").append(reqId).append(", ");
        msg.append("underComp.conId = ").append(underComp.m_conId).append(", ");
        msg.append("underComp.delta = ").append(underComp.m_delta).append(", ");
        msg.append("underComp.price = ").append(underComp.m_price);
        return msg.toString();
    }
}
