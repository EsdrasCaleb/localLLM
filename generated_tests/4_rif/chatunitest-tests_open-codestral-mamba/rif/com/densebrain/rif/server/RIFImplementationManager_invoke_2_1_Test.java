package com.densebrain.rif.server;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.rmi.RemoteException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Hashtable;

@ExtendWith(MockitoExtension.class)
public class RIFImplementationManager_invoke_2_1_Test {

    @Mock
    private RIFImplementationManager manager;

    @Mock
    private Object mockImpl;

    @Mock
    private Method mockMethod;

    @BeforeEach
    public void setUp() throws Exception {
        Map<String, Method> methodMap = new HashMap<>();
        methodMap.put("testMethod", mockMethod);
        Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
        methodsMapField.setAccessible(true);
        Map<Object, Map<String, Method>> methodsMap = (Map<Object, Map<String, Method>>) methodsMapField.get(manager);
        methodsMap.put(mockImpl, methodMap);
    }

    @Test
    public void testInvoke() throws Exception {
        when(mockMethod.invoke(mockImpl, new Object[] { "param1", "param2" })).thenReturn("result");
        Object result = manager.invoke(mockImpl.getClass().getName(), "testMethod", new Object[] { "param1", "param2" });
        assertEquals("result", result);
        verify(mockMethod).invoke(mockImpl, new Object[] { "param1", "param2" });
    }

    @Test
    public void testInvoke_RemoteException_WhenImplNotFound() {
        assertThrows(RemoteException.class, () -> manager.invoke("unknownInterface", "testMethod", new Object[] {}));
    }

    @Test
    public void testInvoke_RemoteException_WhenMethodNotFound() throws Exception {
        Field methodsMapField = RIFImplementationManager.class.getDeclaredField("methodsMap");
        methodsMapField.setAccessible(true);
        Map<Object, Map<String, Method>> methodsMap = (Map<Object, Map<String, Method>>) methodsMapField.get(manager);
        methodsMap.put(mockImpl, new HashMap<>());
        assertThrows(IllegalArgumentException.class, () -> manager.invoke(mockImpl.getClass().getName(), "unknownMethod", new Object[] {}));
    }

    @Test
    public void testInvoke_RemoteException_WhenErrorOccurs() throws Exception {
        when(mockMethod.invoke(mockImpl, new Object[] { "param1", "param2" })).thenThrow(new Exception("error"));
        assertThrows(RemoteException.class, () -> manager.invoke(mockImpl.getClass().getName(), "testMethod", new Object[] { "param1", "param2" }));
    }
}
