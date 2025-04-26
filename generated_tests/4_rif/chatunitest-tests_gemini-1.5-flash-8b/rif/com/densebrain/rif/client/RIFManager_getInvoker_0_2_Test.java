package com.densebrain.rif.client;

import java.rmi.RemoteException;
import java.util.Hashtable;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.RIFServiceStub;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RIFManager_getInvoker_0_2_Test {

    @Mock
    private RIFService mockService;

    @Mock
    private RIFClassLoader mockClassLoader;

    @InjectMocks
    private RIFManager manager;

    @Test
    void testGetInvoker_existingInvoker() throws RemoteException {
        RIFInvoker mockInvoker = mock(RIFInvoker.class);
        try {
            java.lang.reflect.Field invokerMapField = RIFManager.class.getDeclaredField("invokerMap");
            invokerMapField.setAccessible(true);
            // Initialize the map
            Hashtable<Class<?>, RIFInvoker> invokerMap = new Hashtable<>();
            invokerMap.put(String.class, mockInvoker);
            // Correctly set the map
            invokerMapField.set(manager, invokerMap);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        manager = new RIFManager("testUrl") {

            @Override
            protected RIFClassLoader getClassLoader() {
                return mockClassLoader;
            }

            @Override
            public RIFService getService() {
                return mockService;
            }
        };
        RIFInvoker invoker = manager.getInvoker(String.class);
        assertEquals(mockInvoker, invoker);
        verifyNoMoreInteractions(mockService);
        verifyNoMoreInteractions(mockClassLoader);
    }

    @Test
    void testGetInvoker_newInvoker() throws RemoteException {
        manager = new RIFManager("testUrl") {

            @Override
            protected RIFClassLoader getClassLoader() {
                return mockClassLoader;
            }

            @Override
            public RIFService getService() {
                return mockService;
            }
        };
        Class<?> testInterface = String.class;
        RIFInvoker invoker = manager.getInvoker(testInterface);
        assertNotNull(invoker);
        try {
            java.lang.reflect.Field invokerMapField = RIFManager.class.getDeclaredField("invokerMap");
            invokerMapField.setAccessible(true);
            Hashtable<Class<?>, RIFInvoker> invokerMap = (Hashtable<Class<?>, RIFInvoker>) invokerMapField.get(manager);
            assertTrue(invokerMap.containsKey(testInterface));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Error accessing private field: " + e.getMessage());
        }
        verifyNoInteractions(mockService);
        verifyNoInteractions(mockClassLoader);
    }
}
