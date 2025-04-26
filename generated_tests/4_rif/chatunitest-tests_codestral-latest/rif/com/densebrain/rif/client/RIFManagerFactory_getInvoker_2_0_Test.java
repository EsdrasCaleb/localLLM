package com.densebrain.rif.client;

import java.rmi.RemoteException;
import java.util.Hashtable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class RIFManagerFactory_getInvoker_2_0_Test {

    @Mock
    private Hashtable<String, RIFManager> managerMap;

    @InjectMocks
    private RIFManagerFactory rifManagerFactory;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetInvoker() throws RemoteException {
        String url = "testUrl";
        Class interfaceClazz = RIFInvoker.class;
        RIFManager rifManager = mock(RIFManager.class);
        RIFInvoker rifInvoker = mock(RIFInvoker.class);
        when(managerMap.get(url)).thenReturn(rifManager);
        when(rifManager.getInvoker(interfaceClazz)).thenReturn(rifInvoker);
        RIFInvoker result = rifManagerFactory.getInvoker(url, interfaceClazz);
        verify(managerMap).get(url);
        verify(rifManager).getInvoker(interfaceClazz);
        assert result == rifInvoker;
    }
}
