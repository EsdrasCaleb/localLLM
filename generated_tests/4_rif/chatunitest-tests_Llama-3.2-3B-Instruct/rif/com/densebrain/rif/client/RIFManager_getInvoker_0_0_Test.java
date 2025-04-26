package com.densebrain.rif.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
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
    private RIFManager rifManager;

    @Test
    public void testGetInvoker() throws Exception {
        // Arrange
        Map<Class, RIFInvoker> invokerMap = new HashMap<>();
        RIFInvoker invoker = new RIFInvoker(rifManager, RIFInvoker.class);
        invokerMap.put(RIFInvoker.class, invoker);
        when(rifManager.getService()).thenReturn(service);
        when(rifManager.getClassLoader()).thenReturn(classLoader);
        // Act
        RIFInvoker result = rifManager.getInvoker(RIFInvoker.class);
        // Assert
        assertNotNull(result);
        assertEquals(invoker, result);
        assertEquals(1, invokerMap.size());
    }

    @Test
    public void testGetInvokerNotFound() throws Exception {
        // Arrange
        Map<Class, RIFInvoker> invokerMap = new HashMap<>();
        when(rifManager.getService()).thenReturn(service);
        when(rifManager.getClassLoader()).thenReturn(classLoader);
        // Act
        RIFInvoker result = rifManager.getInvoker(RIFInvoker.class);
        // Assert
        assertNotNull(result);
        assertEquals(null, invokerMap.get(RIFInvoker.class));
    }

    @Test
    public void testGetInvokerMultipleThreads() throws Exception {
        // Arrange
        Map<Class, RIFInvoker> invokerMap = new HashMap<>();
        RIFInvoker invoker = new RIFInvoker(rifManager, RIFInvoker.class);
        invokerMap.put(RIFInvoker.class, invoker);
        when(rifManager.getService()).thenReturn(service);
        when(rifManager.getClassLoader()).thenReturn(classLoader);
        // Act and Assert
        RIFInvoker result1 = rifManager.getInvoker(RIFInvoker.class);
        RIFInvoker result2 = rifManager.getInvoker(RIFInvoker.class);
        assertNotNull(result1);
        assertNotNull(result2);
        assertEquals(invoker, result1);
        assertEquals(invoker, result2);
        assertEquals(1, invokerMap.size());
    }
}
