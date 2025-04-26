package com.ib.client;

import java.lang.reflect.Field;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_openOrder_7_1_Test {

    @Mock
    private Contract contract;

    @Mock
    private Order order;

    @Mock
    private OrderState orderState;

    @Mock
    private UnderComp underComp;

    @Mock
    private TagValue tagValue;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }
}
