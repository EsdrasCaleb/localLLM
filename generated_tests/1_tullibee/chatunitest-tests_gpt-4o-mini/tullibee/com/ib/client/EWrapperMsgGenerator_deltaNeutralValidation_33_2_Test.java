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

class EWrapperMsgGenerator_deltaNeutralValidation_33_2_Test {

    @Test
    void testDeltaNeutralValidation() throws Exception {
        int reqId = 123;
        UnderComp underComp = new UnderComp();
        // Using reflection to access private fields
        java.lang.reflect.Field conIdField = UnderComp.class.getDeclaredField("m_conId");
        java.lang.reflect.Field deltaField = UnderComp.class.getDeclaredField("m_delta");
        java.lang.reflect.Field priceField = UnderComp.class.getDeclaredField("m_price");
        conIdField.setAccessible(true);
        deltaField.setAccessible(true);
        priceField.setAccessible(true);
        conIdField.set(underComp, 456);
        deltaField.set(underComp, 0.75);
        priceField.set(underComp, 100.5);
        // Act
        String result = EWrapperMsgGenerator.deltaNeutralValidation(reqId, underComp);
        // Assert
        String expected = "id = 123 underComp.conId =456 underComp.delta =0.75 underComp.price =100.5";
        assertEquals(expected, result);
    }
}
