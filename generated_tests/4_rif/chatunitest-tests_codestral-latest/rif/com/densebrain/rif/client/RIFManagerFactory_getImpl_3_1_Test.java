package com.densebrain.rif.client;

import java.rmi.RemoteException;
import java.util.Hashtable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class RIFManagerFactory_getImpl_3_1_Test {

    @Mock
    private Hashtable<String, RIFManager> managerMap;

    @InjectMocks
    private RIFManagerFactory rifManagerFactory;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetImpl() throws RemoteException {
        String url = "testUrl";
        Class interfaceClazz = RIFManager.class;
        RIFManager rifManager = mock(RIFManager.class);
        RIFInvoker rifInvoker = mock(RIFInvoker.class);
        when(managerMap.get(url)).thenReturn(rifManager);
        when(rifManager.getInvoker(interfaceClazz)).thenReturn(rifInvoker);
        when(rifInvoker.getImpl()).thenReturn(new Object());
        Object result = rifManagerFactory.getImpl(url, interfaceClazz);
        assertNotNull(result);
        verify(managerMap, times(1)).get(url);
        verify(rifManager, times(1)).getInvoker(interfaceClazz);
        verify(rifInvoker, times(1)).getImpl();
    }
}
