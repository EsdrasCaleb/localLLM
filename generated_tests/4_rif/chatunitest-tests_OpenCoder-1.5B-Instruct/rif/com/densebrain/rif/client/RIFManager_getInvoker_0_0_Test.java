package com.densebrain.rif.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.rmi.RemoteException;
import java.util.Hashtable;
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.RIFServiceStub;

@ExtendWith(MockitoExtension.class)
public class RIFManager_getInvoker_0_0_Test {

    @Mock
    private RIFService service;

    @Mock
    private RIFClassLoader classLoader;

    @InjectMocks
    private RIFManager manager;

    @Test
    public void testGetInvoker() throws Exception {
        Class<?> interfaceClass = RIFManager.class;
        Method method = interfaceClass.getDeclaredMethod("getInvoker", Class.class);
        method.setAccessible(true);
        RIFInvoker invoker = mock(RIFInvoker.class);
        when(manager.getInvoker(interfaceClass)).thenReturn(invoker);
        RIFInvoker result = (RIFInvoker) method.invoke(manager, interfaceClass);
        assertSame(invoker, result);
        verify(manager, times(1)).getInvoker(interfaceClass);
    }
}
