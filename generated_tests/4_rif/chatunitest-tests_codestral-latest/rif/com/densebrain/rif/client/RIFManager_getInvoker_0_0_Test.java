package com.densebrain.rif.client;

import java.lang.reflect.Field;
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
public class RIFManager_getInvoker_0_0_Test {

    @InjectMocks
    private RIFManager rifManager;

    @Mock
    private RIFService rifService;

    @Mock
    private RIFClassLoader rifClassLoader;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field invokerMapField = RIFManager.class.getDeclaredField("invokerMap");
        invokerMapField.setAccessible(true);
        invokerMapField.set(rifManager, new Hashtable<Class, RIFInvoker>());
    }

    @Test
    public void testGetInvoker_NewInvoker() throws RemoteException {
        Class<?> interfaceClazz = RIFService.class;
        RIFInvoker invoker = rifManager.getInvoker(interfaceClazz);
        assertNotNull(invoker);
        assertEquals(invoker, rifManager.getInvoker(interfaceClazz));
    }

    @Test
    public void testGetInvoker_ExistingInvoker() throws RemoteException, NoSuchFieldException, IllegalAccessException {
        Class<?> interfaceClazz = RIFService.class;
        RIFInvoker invoker = new RIFInvoker(rifManager, interfaceClazz);
        Field invokerMapField = RIFManager.class.getDeclaredField("invokerMap");
        invokerMapField.setAccessible(true);
        Hashtable<Class, RIFInvoker> invokerMap = (Hashtable<Class, RIFInvoker>) invokerMapField.get(rifManager);
        invokerMap.put(interfaceClazz, invoker);
        RIFInvoker retrievedInvoker = rifManager.getInvoker(interfaceClazz);
        assertNotNull(retrievedInvoker);
        assertEquals(invoker, retrievedInvoker);
    }
}
