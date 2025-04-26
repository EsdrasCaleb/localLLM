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

public class EWrapperMsgGenerator_contractDetailsEnd_18_0_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testContractDetailsEnd() throws Exception {
        int reqId = 123;
        String expected = "reqId = 123 =============== end ===============";
        // Use reflection to invoke the private method
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("contractDetailsEnd", int.class);
        method.setAccessible(true);
        String result = (String) method.invoke(eWrapperMsgGenerator, reqId);
        assertEquals(expected, result);
    }
}
